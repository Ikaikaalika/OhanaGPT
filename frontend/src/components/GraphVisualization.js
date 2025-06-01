import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

const GraphVisualization = ({ fileId, highlightIndividual, predictions }) => {
  const svgRef = useRef(null);
  const [graphData, setGraphData] = useState(null);

  useEffect(() => {
    // In a real implementation, this would fetch graph data from the API
    // For now, we'll create sample data
    const sampleData = createSampleGraphData(highlightIndividual, predictions);
    setGraphData(sampleData);
  }, [fileId, highlightIndividual, predictions]);

  useEffect(() => {
    if (!graphData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const width = 800;
    const height = 600;

    svg.attr('width', width).attr('height', height);

    const simulation = d3.forceSimulation(graphData.nodes)
      .force('link', d3.forceLink(graphData.links).id(d => d.id).distance(50))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2));

    const g = svg.append('g');

    // Add zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 3])
      .on('zoom', (event) => {
        g.attr('transform', event.transform);
      });

    svg.call(zoom);

    // Create arrow markers for directed edges
    svg.append('defs').append('marker')
      .attr('id', 'arrowhead')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 25)
      .attr('refY', 0)
      .attr('markerWidth', 5)
      .attr('markerHeight', 5)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#999');

    // Draw links
    const link = g.append('g')
      .selectAll('line')
      .data(graphData.links)
      .enter().append('line')
      .attr('class', 'link')
      .attr('stroke', d => d.type === 'predicted' ? '#ff6b6b' : '#999')
      .attr('stroke-width', d => d.type === 'predicted' ? 3 : 1)
      .attr('stroke-dasharray', d => d.type === 'predicted' ? '5,5' : '')
      .attr('marker-end', 'url(#arrowhead)');

    // Draw nodes
    const node = g.append('g')
      .selectAll('g')
      .data(graphData.nodes)
      .enter().append('g')
      .attr('class', 'node')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    node.append('circle')
      .attr('r', d => d.id === highlightIndividual ? 15 : 10)
      .attr('fill', d => {
        if (d.id === highlightIndividual) return '#4ecdc4';
        if (d.gender === 'M') return '#3498db';
        if (d.gender === 'F') return '#e74c3c';
        return '#95a5a6';
      })
      .attr('stroke', d => d.isPredicted ? '#ff6b6b' : '#fff')
      .attr('stroke-width', d => d.isPredicted ? 3 : 1);

    node.append('text')
      .attr('dx', 12)
      .attr('dy', 4)
      .text(d => d.name || d.id)
      .style('font-size', '12px');

    // Add tooltips
    node.append('title')
      .text(d => `${d.name || d.id}\nBorn: ${d.birthYear || 'Unknown'}`);

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => d.source.x)
        .attr('y1', d => d.source.y)
        .attr('x2', d => d.target.x)
        .attr('y2', d => d.target.y);

      node.attr('transform', d => `translate(${d.x},${d.y})`);
    });

    function dragstarted(event, d) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event, d) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event, d) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }
  }, [graphData, highlightIndividual]);

  const createSampleGraphData = (targetId, predictions) => {
    // Create sample visualization data
    const nodes = [
      { id: targetId, name: 'Target Individual', gender: 'U' }
    ];
    
    const links = [];

    if (predictions && predictions.predictions) {
      predictions.predictions.slice(0, 5).forEach((pred, index) => {
        nodes.push({
          id: pred.id,
          name: pred.name,
          gender: pred.gender,
          birthYear: pred.birth_year,
          isPredicted: true
        });
        links.push({
          source: pred.id,
          target: targetId,
          type: 'predicted',
          confidence: pred.score
        });
      });
    }

    return { nodes, links };
  };

  return (
    <div className="graph-visualization">
      <svg ref={svgRef}></svg>
      <div className="visualization-legend">
        <div className="legend-item">
          <span className="legend-color male"></span> Male
        </div>
        <div className="legend-item">
          <span className="legend-color female"></span> Female
        </div>
        <div className="legend-item">
          <span className="legend-color predicted"></span> Predicted Parent
        </div>
      </div>
    </div>
  );
};

export default GraphVisualization;